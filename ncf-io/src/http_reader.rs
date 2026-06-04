#![cfg(feature = "http")]

use bytes::Bytes;
use futures::stream::{self, BoxStream, StreamExt};
use reqwest::header::RANGE;
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
}

impl NcfHttpReader {
    /// Open a remote NCF file using two range requests.
    pub async fn open(url: &str) -> Result<Self> {
        let client = Client::new();

        // 1) fetch prefix
        let prefix_bytes = client
            .get(url)
            .header(RANGE, "bytes=0-47")
            .send()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
            .bytes()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        let prefix = FileHeaderPrefix::decode(&prefix_bytes)?;

        // 2) fetch rest of file from byte 48 to end
        let range_header = format!("bytes={}-", FILE_HEADER_PREFIX_SIZE);
        let body = client
            .get(url)
            .header(RANGE, range_header)
            .send()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
            .bytes()
            .await
            .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        let file_size = (FILE_HEADER_PREFIX_SIZE as u64)
            .checked_add(body.len() as u64)
            .ok_or_else(|| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, "file size overflow")))?;

        let header_len = prefix.header_len as usize;
        if body.len() < header_len {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "body too small for header")));
        }

        let metadata = NcfHeader::decode_cbor(&body[..header_len])?;

        let schema_rel_start = (prefix.schema_offset as usize).saturating_sub(FILE_HEADER_PREFIX_SIZE as usize);
        let schema_rel_end = (prefix.index_offset as usize).saturating_sub(FILE_HEADER_PREFIX_SIZE as usize);
        let schema_bytes = &body[schema_rel_start..schema_rel_end];
        let schemas: Vec<TensorSchema> = from_reader(Cursor::new(schema_bytes))?;

        let footer_pos = body.len().checked_sub(16).ok_or_else(|| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "file too small for footer")))?;
        let footer_magic = &body[footer_pos..footer_pos + 8];
        if footer_magic != b"NCFEND!!" {
            return Err(ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid footer magic")));
        }

        let footer_len_bytes: [u8; 8] = body[footer_pos + 8..footer_pos + 16]
            .try_into()
            .map_err(|_| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid footer len")))?;
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
        .ok_or_else(|| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, "chunk range overflow")))?;
    let range_header = format!("bytes={}-{}", start, end - 1);

    let bytes = client
        .get(url)
        .header(RANGE, range_header)
        .send()
        .await
        .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
        .bytes()
        .await
        .map_err(|e| ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

    let mut cursor = Cursor::new(&bytes);
    let mut header_buf = [0u8; CHUNK_HEADER_SIZE as usize];
    cursor.read_exact(&mut header_buf).map_err(|e| ncf_core::NcfError::Io(e))?;
    let chunk_header = ChunkHeader::decode(&header_buf)?;

    let mut payload = vec![0u8; chunk_header.compressed_len as usize];
    cursor.read_exact(&mut payload).map_err(|e| ncf_core::NcfError::Io(e))?;

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
        return Err(ncf_core::NcfError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, "checksum mismatch")));
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
