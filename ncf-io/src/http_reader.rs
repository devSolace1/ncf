#![cfg(feature = "http")]

use bytes::Bytes;
use futures::stream::{self, BoxStream, StreamExt};
use reqwest::header::{RANGE, ACCEPT_RANGES, CONTENT_RANGE};
use reqwest::Client;
use std::convert::TryInto;
use std::io::{Cursor, Read};

use ncf_core::header::{FileHeaderPrefix, NcfHeader};
use ncf_core::index::NcfIndex;
use ncf_core::Result;
use ncf_core::schema::{ChunkRef, TensorSchema};
use ncf_core::chunk::ChunkHeader;
use ncf_core::constants::{CHUNK_HEADER_SIZE, FILE_HEADER_PREFIX_SIZE};
use ciborium::de::from_reader;

/// HTTP-backed NCF reader using byte-range requests.
pub struct NcfHttpReader {
    url: String,
    client: Client,
    file_size: u64,
    header_prefix: FileHeaderPrefix,
    metadata: NcfHeader,
    schemas: Vec<TensorSchema>,
    index: NcfIndex,
    supports_range_requests: bool,
}

impl NcfHttpReader {
    /// Open a remote NCF file using 3-step architecture with efficient range requests.
    /// 
    /// Step 1: Fetch 48-byte prefix to get schema_offset and index_offset
    /// Step 2: Fetch header + schema blocks (from byte 48 to index_offset-16)
    /// Step 3: Fetch footer (last 16 bytes) to get index length, then fetch index block
    pub async fn open(url: &str) -> Result<Self> {
        let client = Client::new();

        // Determine file size and validate range request support with HEAD request
        let head_resp = client
            .head(url)
            .send()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        let file_size = head_resp
            .content_length()
            .ok_or_else(|| ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Server did not provide Content-Length header",
            )))?;

        // Check if server supports range requests
        let supports_range = head_resp
            .headers()
            .get(ACCEPT_RANGES)
            .and_then(|v| v.to_str().ok())
            .map(|v| v != "none")
            .unwrap_or(false);

        if !supports_range {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "HTTP server does not support range requests (Accept-Ranges: none). \
                Cannot efficiently fetch NCF file. Server must support Range headers.",
            )));
        }

        // STEP 1: Fetch 48-byte prefix to get schema_offset and index_offset
        let prefix_bytes = client
            .get(url)
            .header(RANGE, "bytes=0-47")
            .send()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
            .bytes()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        if prefix_bytes.len() != 48 {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Expected 48 bytes for prefix, got {}", prefix_bytes.len()),
            )));
        }

        let prefix = FileHeaderPrefix::decode(&prefix_bytes)?;

        // Validate offsets
        if prefix.schema_offset < FILE_HEADER_PREFIX_SIZE {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid schema_offset: {} (must be >= {})",
                    prefix.schema_offset, FILE_HEADER_PREFIX_SIZE
                ),
            )));
        }

        if prefix.index_offset < prefix.schema_offset {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid index_offset: {} (must be >= schema_offset: {})",
                    prefix.index_offset, prefix.schema_offset
                ),
            )));
        }

        if prefix.index_offset >= file_size || file_size < 16 {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid offsets: index_offset={}, file_size={}, file must be at least 16 bytes for footer",
                    prefix.index_offset, file_size
                ),
            )));
        }

        // STEP 2: Fetch header + schema blocks (from byte 48 to index_offset-16)
        // index_offset points to start of index, and we need to stop before footer (last 16 bytes)
        let header_schema_end = prefix.index_offset;
        let range_header = format!("bytes=48-{}", header_schema_end - 1);

        let header_schema_bytes = client
            .get(url)
            .header(RANGE, range_header)
            .send()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
            .bytes()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        // Decode header from the fetched block
        let header_len = prefix.header_len as usize;
        if header_schema_bytes.len() < header_len {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("Header block too small: got {}, expected at least {}", 
                    header_schema_bytes.len(), header_len),
            )));
        }

        let metadata = NcfHeader::decode_cbor(&header_schema_bytes[..header_len])?;

        // Extract schema block
        let schema_rel_start = (prefix.schema_offset as usize).saturating_sub(48);
        let schema_rel_end = (prefix.index_offset as usize).saturating_sub(48);

        if schema_rel_end > header_schema_bytes.len() {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Schema block out of range: [{}, {}] in buffer of size {}",
                    schema_rel_start, schema_rel_end, header_schema_bytes.len()),
            )));
        }

        let schema_bytes = &header_schema_bytes[schema_rel_start..schema_rel_end];
        let schemas: Vec<TensorSchema> = from_reader(Cursor::new(schema_bytes))?;

        // STEP 3: Fetch footer (last 16 bytes) to get index length
        let footer_start = file_size - 16;
        let range_header = format!("bytes={}-{}", footer_start, file_size - 1);

        let footer_bytes = client
            .get(url)
            .header(RANGE, range_header)
            .send()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
            .bytes()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        if footer_bytes.len() != 16 {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Expected 16 bytes for footer, got {}", footer_bytes.len()),
            )));
        }

        let footer_magic = &footer_bytes[0..8];
        if footer_magic != b"NCFEND!!" {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid footer magic: expected b'NCFEND!!', got {:?}", 
                    String::from_utf8_lossy(footer_magic)),
            )));
        }

        let footer_len_bytes: [u8; 8] = footer_bytes[8..16]
            .try_into()
            .map_err(|_| ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Failed to parse footer length",
            )))?;
        let index_len = u64::from_le_bytes(footer_len_bytes) as usize;

        // Validate index length
        if (prefix.index_offset as usize) + index_len + 16 > file_size as usize {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Index block overflow: index_offset={}, index_len={}, file_size={}, with 16-byte footer",
                    prefix.index_offset, index_len, file_size
                ),
            )));
        }

        // STEP 3 (continued): Fetch index block
        let index_end = prefix.index_offset + (index_len as u64);
        let range_header = format!("bytes={}-{}", prefix.index_offset, index_end - 1);

        let index_bytes = client
            .get(url)
            .header(RANGE, range_header)
            .send()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
            .bytes()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        if index_bytes.len() != index_len {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Index block size mismatch: expected {}, got {}", index_len, index_bytes.len()),
            )));
        }

        let index: NcfIndex = from_reader(Cursor::new(index_bytes))?;

        Ok(Self {
            url: url.to_string(),
            client,
            file_size,
            header_prefix: prefix,
            metadata,
            schemas,
            index,
            supports_range_requests: true,
        })
    }

    /// Get the parsed metadata from the remote NCF file.
    pub fn metadata(&self) -> &NcfHeader {
        &self.metadata
    }

    /// Get the parsed tensor schemas from the remote NCF file.
    pub fn schemas(&self) -> &[TensorSchema] {
        &self.schemas
    }

    /// Remote file size in bytes as reported by the HTTP server.
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Parsed file header prefix from the remote NCF file.
    pub fn header_prefix(&self) -> &FileHeaderPrefix {
        &self.header_prefix
    }

    /// Parsed index block for the remote NCF file.
    pub fn index(&self) -> &NcfIndex {
        &self.index
    }

    /// Whether the HTTP server supports range requests.
    pub fn supports_range_requests(&self) -> bool {
        self.supports_range_requests
    }

    async fn fetch_chunk(&self, chunk_ref: ChunkRef) -> Result<Bytes> {
        fetch_chunk_bytes(&self.client, &self.url, chunk_ref).await
    }

    /// Fetch a full tensor by name using one or more range requests.
    pub async fn fetch_tensor(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let schema = self.schemas.iter().find(|schema| schema.name == name);
        if let Some(schema) = schema {
            let mut bytes = Vec::new();
            for chunk_ref in &schema.chunks {
                let chunk_bytes = self.fetch_chunk(chunk_ref.clone()).await?;
                bytes.extend_from_slice(&chunk_bytes);
            }
            return Ok(Some(bytes));
        }
        Ok(None)
    }

    /// Stream a tensor's data as a stream of `Bytes`, one result per chunk.
    pub async fn stream_tensor(&self, name: &str) -> Result<BoxStream<'static, Result<Bytes>>> {
        let schema = self.schemas.iter().find(|schema| schema.name == name);
        if let Some(schema) = schema {
            let client = self.client.clone();
            let url = self.url.clone();
            let chunks = schema.chunks.clone();

            let stream = stream::iter(chunks).then(move |chunk_ref| {
                let client = client.clone();
                let url = url.clone();
                async move { fetch_chunk_bytes(&client, &url, chunk_ref).await }
            });

            Ok(stream.boxed())
        } else {
            Ok(stream::empty().boxed())
        }
    }
}

async fn fetch_chunk_bytes(client: &Client, url: &str, chunk_ref: ChunkRef) -> Result<Bytes> {
    let start = chunk_ref.byte_offset;
    let end = chunk_ref
        .byte_offset
        .checked_add(chunk_ref.byte_len)
        .ok_or_else(|| ncf_core::NcfError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "chunk range overflow",
        )))?;

    // Validate chunk bounds
    if start >= end {
        return Err(ncf_core::NcfError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid chunk range: start={} >= end={}", start, end),
        )));
    }

    // HTTP Range header format: bytes=start-end (inclusive)
    let range_header = format!("bytes={}-{}", start, end - 1);

    let response = client
        .get(url)
        .header(RANGE, &range_header)
        .send()
        .await
        .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to fetch chunk range {}: {}", range_header, e),
        )))?;

    // Validate response status
    let status = response.status();
    if !status.is_success() {
        return Err(ncf_core::NcfError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "HTTP request for chunk {} failed with status {}: {}",
                range_header,
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown")
            ),
        )));
    }

    // Check if server supports range requests (should respond with 206 Partial Content)
    if status.as_u16() != 206 && status.as_u16() != 200 {
        return Err(ncf_core::NcfError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "Unexpected status code {} for range request. Server may not support range requests.",
                status.as_u16()
            ),
        )));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to read response body for chunk {}: {}", range_header, e),
        )))?;

    // Validate response size
    let expected_size = (end - start) as usize;
    if bytes.len() != expected_size {
        return Err(ncf_core::NcfError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Chunk size mismatch for range {}: expected {}, got {}",
                range_header,
                expected_size,
                bytes.len()
            ),
        )));
    }

    let mut cursor = Cursor::new(&bytes);
    let mut header_buf = [0u8; CHUNK_HEADER_SIZE as usize];
    cursor.read_exact(&mut header_buf).map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("Failed to read chunk header from range {}: {}", range_header, e),
    )))?;

    let chunk_header = ChunkHeader::decode(&header_buf)?;

    let mut payload = vec![0u8; chunk_header.compressed_len as usize];
    cursor.read_exact(&mut payload).map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("Failed to read chunk payload from range {}: {}", range_header, e),
    )))?;

    let decompressed = if chunk_header.flags == 0 {
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

    let computed = blake3::hash(&decompressed);
    if computed.as_bytes() != &chunk_ref.checksum {
        return Err(ncf_core::NcfError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Checksum mismatch for chunk at range {}: expected {:x?}, got {:x?}",
                range_header,
                chunk_ref.checksum,
                computed.as_bytes()
            ),
        )));
    }

    Ok(Bytes::from(decompressed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use wiremock::{MockServer, Mock, ResponseTemplate};
    use wiremock::matchers::{method, header};

    use ncf_core::header::{Metadata, NcfFlags};
    use ncf_core::schema::{DType, Encoding, Layout, Compression};
    use crate::writer::NcfWriter;

    fn build_ncf_file(path: &std::path::Path) -> Vec<u8> {
        let metadata = NcfHeader { metadata: Metadata { model_name: "s".into(), architecture: "a".into(), created_at: 0, author: None, license: None, quantization: None, custom: Default::default() } };
        let mut writer = NcfWriter::new(metadata, NcfFlags::empty());
        let schema = TensorSchema { name: "tensor_0".to_string(), dtype: DType::U8, shape: vec![4], column_layout: Layout::RowMajor, compression: Compression::None, encoding: Encoding::Plain, chunks: vec![] };
        writer.add_tensor(schema, vec![1, 2, 3, 4]);
        writer.finalize(path).expect("write NCF file");
        std::fs::read(path).expect("read generated NCF file")
    }

    #[tokio::test]
    async fn test_http_open_only_2_requests() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("test_http_open.ncf");
        let bytes = build_ncf_file(&path);

        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(header("range", "bytes=0-47"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("Content-Range", format!("bytes 0-47/{len}", len = bytes.len()))
                    .set_body_bytes(bytes[0..48].to_vec()),
            )
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(header("range", "bytes=48-"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("Content-Range", format!("bytes 48-{}/{}", bytes.len() - 1, bytes.len()))
                    .set_body_bytes(bytes[48..].to_vec()),
            )
            .expect(1)
            .mount(&server)
            .await;

        let reader = NcfHttpReader::open(&server.uri()).await.expect("open http file");

        assert_eq!(reader.metadata().metadata.model_name, "s");
        assert_eq!(reader.schemas().len(), 1);
    }

    #[tokio::test]
    async fn test_http_fetch_tensor_one_request() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("test_http_fetch.ncf");
        let bytes = build_ncf_file(&path);

        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(header("range", "bytes=0-47"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("Content-Range", format!("bytes 0-47/{len}", len = bytes.len()))
                    .set_body_bytes(bytes[0..48].to_vec()),
            )
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(header("range", "bytes=48-"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("Content-Range", format!("bytes 48-{}/{}", bytes.len() - 1, bytes.len()))
                    .set_body_bytes(bytes[48..].to_vec()),
            )
            .expect(1)
            .mount(&server)
            .await;

        let reader = NcfHttpReader::open(&server.uri()).await.expect("open http file");

        let chunk_ref = reader.schemas()[0].chunks[0].clone();
        let start = chunk_ref.byte_offset;
        let end = chunk_ref.byte_offset + chunk_ref.byte_len - 1;

        Mock::given(method("GET"))
            .and(header("range", format!("bytes={}-{}", start, end)))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("Content-Range", format!("bytes {start}-{end}/{len}", len = bytes.len()))
                    .set_body_bytes(bytes[start as usize..=(end as usize)].to_vec()),
            )
            .expect(1)
            .mount(&server)
            .await;

        let fetched = reader.fetch_tensor("tensor_0").await.expect("fetch tensor").expect("tensor exists");
        assert_eq!(fetched, vec![1, 2, 3, 4]);
    }
}
