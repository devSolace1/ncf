use ciborium::de::from_reader;
use ncf_core::header::{FileHeaderPrefix, NcfHeader};
use ncf_core::index::NcfIndex;
use ncf_core::schema::TensorSchema;
use ncf_core::Result;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub struct NcfReader {
    pub metadata: NcfHeader,
    pub schemas: Vec<TensorSchema>,
    pub index: NcfIndex,
    pub header_prefix: FileHeaderPrefix,
    file: File,
}

impl NcfReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut prefix_bytes = [0u8; 48];
        file.read_exact(&mut prefix_bytes)?;
        let header_prefix = FileHeaderPrefix::decode(&prefix_bytes)?;

        let mut header_bytes = vec![0u8; header_prefix.header_len as usize];
        file.read_exact(&mut header_bytes)?;
        let metadata = NcfHeader::decode_cbor(&header_bytes)?;

        file.seek(SeekFrom::Start(header_prefix.schema_offset))?;
        let schema_len = (header_prefix.index_offset - header_prefix.schema_offset) as usize;
        let mut schema_block = vec![0u8; schema_len];
        file.read_exact(&mut schema_block)?;
        let schemas: Vec<TensorSchema> = from_reader(std::io::Cursor::new(&schema_block))?;

        file.seek(SeekFrom::Start(header_prefix.index_offset))?;
        let index_len = {
            let mut footer_magic = [0u8; 8];
            let mut footer_len_bytes = [0u8; 8];
            let footer_position = file.metadata()?.len() - 16;
            file.seek(SeekFrom::Start(footer_position))?;
            file.read_exact(&mut footer_magic)?;
            if &footer_magic != b"NCFEND!!" {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing footer magic").into());
            }
            file.read_exact(&mut footer_len_bytes)?;
            u64::from_le_bytes(footer_len_bytes)
        };
        file.seek(SeekFrom::Start(header_prefix.index_offset))?;
        let mut index_block = vec![0u8; index_len as usize];
        file.read_exact(&mut index_block)?;
        let index: NcfIndex = from_reader(std::io::Cursor::new(&index_block))?;

        Ok(Self {
            metadata,
            schemas,
            index,
            header_prefix,
            file,
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

    pub fn find_schema(&self, name: &str) -> Option<&TensorSchema> {
        self.schemas.iter().find(|schema| schema.name == name)
    }

    pub fn read_tensor(&mut self, name: &str) -> Result<Option<Vec<u8>>> {
        let schema = match self.find_schema(name) {
            Some(schema) => schema.clone(),
            None => return Ok(None),
        };

        let mut result = Vec::new();
        for chunk in &schema.chunks {
            self.file.seek(SeekFrom::Start(chunk.byte_offset))?;
            let mut header_bytes = [0u8; 30];
            self.file.read_exact(&mut header_bytes)?;
            let payload_len = (chunk.byte_len as usize).saturating_sub(header_bytes.len() + 32);
            let mut payload = vec![0u8; payload_len];
            self.file.read_exact(&mut payload)?;
            let mut checksum = [0u8; 32];
            self.file.read_exact(&mut checksum)?;
            result.extend_from_slice(&payload);
        }
        Ok(Some(result))
    }
}
