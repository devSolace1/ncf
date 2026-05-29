pub const CHUNK_MAGIC: &[u8; 4] = b"NCFK";

#[derive(Debug, Clone, Copy)]
pub struct ChunkHeader {
    pub chunk_id: u64,
    pub flags: u16,
    pub uncompressed_len: u64,
    pub compressed_len: u64,
}

impl ChunkHeader {
    pub fn encode(&self) -> [u8; 30] {
        let mut bytes = [0u8; 30];
        bytes[..4].copy_from_slice(CHUNK_MAGIC);
        bytes[4..12].copy_from_slice(&self.chunk_id.to_le_bytes());
        bytes[12..14].copy_from_slice(&self.flags.to_le_bytes());
        bytes[14..22].copy_from_slice(&self.uncompressed_len.to_le_bytes());
        bytes[22..30].copy_from_slice(&self.compressed_len.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> std::io::Result<Self> {
        if bytes.len() < 30 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Chunk header too short"));
        }
        if &bytes[..4] != CHUNK_MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid chunk magic"));
        }
        let chunk_id = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
        let flags = u16::from_le_bytes(bytes[12..14].try_into().unwrap());
        let uncompressed_len = u64::from_le_bytes(bytes[14..22].try_into().unwrap());
        let compressed_len = u64::from_le_bytes(bytes[22..30].try_into().unwrap());
        Ok(Self {
            chunk_id,
            flags,
            uncompressed_len,
            compressed_len,
        })
    }
}
