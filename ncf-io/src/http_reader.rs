#![cfg(feature = "http")]

use bytes::Bytes;
use futures::stream::{self, Stream};
use reqwest::header::RANGE;
use reqwest::Client;
use std::convert::TryInto;
use std::sync::Arc;

use ncf_core::header::{FileHeaderPrefix, NcfHeader};
use ncf_core::index::NcfIndex;
use ncf_core::Result;
use ncf_core::schema::TensorSchema;
use ncf_core::chunk::ChunkHeader;
use ncf_core::constants::{CHUNK_HEADER_SIZE, FILE_HEADER_PREFIX_SIZE};
use ciborium::de::from_reader;
use std::io::{Cursor, Read};

/// HTTP-backed NCF reader using byte-range requests.
pub struct NcfHttpReader {
    url: String,
    client: Client,
    file_size: u64,
    header_prefix: FileHeaderPrefix,
    metadata: NcfHeader,
    schemas: Vec<TensorSchema>,
    index: NcfIndex,
}

impl NcfHttpReader {
    /// Open a remote NCF file using two range requests.
    pub async fn open(url: &str) -> Result<Self> {
        let client = Client::new();
        // 1) fetch prefix
        let resp = client
            .get(url)
            .header(RANGE, "bytes=0-47")
            .send()
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let prefix_bytes = resp.bytes().await.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let prefix = FileHeaderPrefix::decode(&prefix_bytes)?;

        // 2) fetch rest of file from byte 48 to end
        let range_header = format!("bytes={}-", FILE_HEADER_PREFIX_SIZE);
        let resp2 = client
            .get(url)
            .header(RANGE, range_header)
            .send()
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let body = resp2.bytes().await.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // file size: prefix + remaining
        let file_size = (FILE_HEADER_PREFIX_SIZE as u64).checked_add(body.len() as u64).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "file size overflow"))?;

        // parse header from start of body
        let header_len = prefix.header_len as usize;
        if body.len() < header_len {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "body too small for header").into());
        }
        let header_bytes = &body[..header_len];
        let metadata = NcfHeader::decode_cbor(header_bytes)?;

        // schema block is at offset (schema_offset - 48) within body
        let schema_rel_start = (prefix.schema_offset as usize).saturating_sub(FILE_HEADER_PREFIX_SIZE as usize);
        let schema_rel_end = (prefix.index_offset as usize).saturating_sub(FILE_HEADER_PREFIX_SIZE as usize);
        let schema_bytes = &body[schema_rel_start..schema_rel_end];
        let schemas: Vec<TensorSchema> = from_reader(Cursor::new(schema_bytes))?;

        // index block is located earlier in the body: starting at (index_offset - 48)
        let footer_pos = body.len() - 16; // last 16 bytes are footer
        let footer_magic = &body[footer_pos..footer_pos + 8];
        if footer_magic != b"NCFEND!!" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid footer magic").into());
        }
        let footer_len_bytes: [u8; 8] = body[footer_pos + 8..footer_pos + 16].try_into().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid footer len"))?;
        let index_len = u64::from_le_bytes(footer_len_bytes) as usize;
        let index_rel_start = (prefix.index_offset as usize).saturating_sub(FILE_HEADER_PREFIX_SIZE as usize);
        let index_bytes = &body[index_rel_start..index_rel_start + index_len];
        let index: NcfIndex = from_reader(Cursor::new(index_bytes))?;

        Ok(Self {
            url: url.to_string(),
            client,
            file_size,
            header_prefix: prefix,
            metadata,
            schemas,
            index,
        })
    }

    /// Fetch a full tensor by name using a single range request (if possible).
    pub async fn fetch_tensor(&self, name: &str) -> Result<Option<Vec<u8>>> {
        if let Some(chunk_id) = self.index.find_chunk_id(name) {
            if let Some(entry) = self.index.chunk_map.get(&chunk_id) {
                let start = entry.byte_offset;
                let end = entry.byte_offset.checked_add(entry.byte_len).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "chunk range overflow"))? - 1;
                let range_header = format!("bytes={}-{}", start, end);
                let resp = self.client.get(&self.url).header(RANGE, range_header).send().await.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                let bytes = resp.bytes().await.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                // parse chunk header and payload
                let mut cur = Cursor::new(bytes);
                let mut hdr = vec![0u8; CHUNK_HEADER_SIZE as usize];
                cur.read_exact(&mut hdr).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                let chunk_header = ChunkHeader::decode(&hdr)?;
                let mut payload = vec![0u8; chunk_header.compressed_len as usize];
                cur.read_exact(&mut payload).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                // decompress if needed
                let data = if chunk_header.flags == 0 {
                    payload
                } else if let Ok(d) = zstd::decode_all(&*payload) {
                    d
                } else if let Ok(d) = lz4_flex::decompress_size_prepended(&payload) {
                    d
                } else if let Ok(d) = snap::raw::Decoder::new().decompress_vec(&payload) {
                    d
                } else {
                    payload
                };
                return Ok(Some(data));
            }
        }
        Ok(None)
    }

    /// Stream a tensor's data as a stream of Bytes (currently single-item stream).
    pub async fn stream_tensor(&self, name: &str) -> Result<impl Stream<Item = Result<Bytes>>> {
        if let Some(data) = self.fetch_tensor(name).await? {
            let bytes = Bytes::from(data);
            let s = stream::once(async move { Ok(bytes) });
            Ok(s)
        } else {
            Ok(stream::empty())
        }
    }
}
