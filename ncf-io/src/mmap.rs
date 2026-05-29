use memmap2::Mmap;
use ncf_core::header::FileHeaderPrefix;
use ncf_core::index::NcfIndex;
use ncf_core::schema::TensorSchema;
use ncf_core::Result;
use std::fs::File;
use std::path::Path;

pub struct NcfMmap {
    pub mmap: Mmap,
    pub header_prefix: FileHeaderPrefix,
    pub metadata: ncf_core::header::NcfHeader,
    pub schemas: Vec<TensorSchema>,
    pub index: NcfIndex,
}

impl NcfMmap {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 48 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "file too small").into());
        }
        let header_prefix = FileHeaderPrefix::decode(&mmap[..48])?;
        let header_start = 48;
        let header_end = header_start + header_prefix.header_len as usize;
        let metadata = ncf_core::header::NcfHeader::decode_cbor(&mmap[header_start..header_end])?;
        let schema_start = header_prefix.schema_offset as usize;
        let schema_end = header_prefix.index_offset as usize;
        let schemas: Vec<TensorSchema> = ciborium::de::from_reader(std::io::Cursor::new(&mmap[schema_start..schema_end]))?;
        let index_end = mmap.len() - 16;
        let footer_magic = &mmap[index_end..index_end + 8];
        if footer_magic != b"NCFEND!!" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing footer magic").into());
        }
        let footer_len = u64::from_le_bytes(mmap[index_end + 8..index_end + 16].try_into().unwrap()) as usize;
        let index_start = header_prefix.index_offset as usize;
        let index_end = index_start + footer_len;
        let index: NcfIndex = ciborium::de::from_reader(std::io::Cursor::new(&mmap[index_start..index_end]))?;
        Ok(Self {
            mmap,
            header_prefix,
            metadata,
            schemas,
            index,
        })
    }

    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let chunk_id = self.index.tensor_map.get(name)?;
        let entry = self.index.entries.iter().find(|entry| &entry.chunk_id == chunk_id)?;
        let offset = entry.byte_offset as usize + 30;
        let end = offset + (entry.byte_len as usize - 30 - 32);
        Some(&self.mmap[offset..end])
    }
}
