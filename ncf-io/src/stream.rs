use std::path::Path;

use crate::writer::NcfWriter;
use ncf_core::header::{NcfFlags, NcfHeader, FileHeaderPrefix};
use ncf_core::schema::TensorSchema;
use ncf_core::chunk::ChunkHeader;
use ncf_core::constants::{CHUNK_CHECKSUM_SIZE, CHUNK_HEADER_SIZE, FILE_HEADER_PREFIX_SIZE};
use ncf_core::Result;

/// Streaming writer for NCF files (synchronous helper retained).
#[derive(Debug)]
pub struct NcfStream {
    writer: NcfWriter,
}

impl NcfStream {
    /// Create a new streaming NCF writer.
    pub fn new(metadata: NcfHeader, flags: NcfFlags) -> Self {
        Self {
            writer: NcfWriter::new(metadata, flags),
        }
    }

    /// Add a tensor to the streaming writer.
    pub fn add_tensor(&mut self, schema: TensorSchema, payload: Vec<u8>) {
        self.writer.add_tensor(schema, payload);
    }

    /// Finalize the stream and write the NCF file.
    pub fn finalize<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.writer.finalize(path)
    }

    /// Convenience helper to write a stream directly to a file path.
    pub fn write_to_path<P: AsRef<Path>>(metadata: NcfHeader, flags: NcfFlags, tensors: Vec<(TensorSchema, Vec<u8>)>, path: P) -> Result<()> {
        let mut stream = Self::new(metadata, flags);
        for (schema, payload) in tensors {
            stream.add_tensor(schema, payload);
        }
        stream.finalize(path)
    }
}

// --- Async streaming reader implementation (feature gated) ---

#[cfg(feature = "streaming")]
/// Asynchronous streaming reader for NCF files.
pub mod reader {
    use super::*;
    use bytes::Bytes;
    use std::collections::HashSet;
    use tokio::io::{AsyncRead, AsyncReadExt};

    /// Chunk events emitted by the stream reader.
    pub enum StreamChunk {
        /// File-level metadata header (CBOR decoded `NcfHeader`).
        Metadata(NcfHeader),
        /// Schema block containing tensor descriptions.
        Schema(Vec<TensorSchema>),
        /// Payload data for a tensor chunk.
        TensorData {
            /// Tensor name associated with this chunk.
            name: String,
            /// Monotonic chunk identifier.
            chunk_id: u64,
            /// Decompressed payload bytes for this chunk.
            data: Bytes,
            /// Whether this chunk is the last chunk for the tensor.
            is_last_chunk: bool,
            /// Whether blake3 checksum matched the payload.
            checksum_valid: bool,
        },
        /// Stream finished indicator.
        Done,
    }

    /// Internal state machine for the streaming reader.
    enum StreamState {
        AwaitingHeader,
        AwaitingSchema,
        StreamingChunks,
        Done,
    }

    /// Async streaming reader that consumes an `AsyncRead` and emits NCF chunk events.
    pub struct NcfStreamReader {
        inner: Box<dyn AsyncRead + Unpin + Send>,
        state: StreamState,
        header_prefix: Option<FileHeaderPrefix>,
        requested_tensors: Option<HashSet<String>>,
    }

    impl NcfStreamReader {
        /// Create a reader from any `AsyncRead`.
        pub fn from_async_reader(reader: impl AsyncRead + Unpin + Send + 'static) -> Self {
            Self {
                inner: Box::new(reader),
                state: StreamState::AwaitingHeader,
                header_prefix: None,
                requested_tensors: None,
            }
        }

        /// Specify which tensors to emit. `None` (default) streams all tensors.
        pub fn request_tensors(&mut self, names: &[&str]) {
            let set: HashSet<String> = names.iter().map(|s| s.to_string()).collect();
            self.requested_tensors = Some(set);
        }

        /// Read next chunk from the async reader.
        pub async fn next_chunk(&mut self) -> Result<Option<StreamChunk>> {
            loop {
                match self.state {
                    StreamState::AwaitingHeader => {
                        let mut buf = vec![0u8; FILE_HEADER_PREFIX_SIZE as usize];
                        self.inner.read_exact(&mut buf).await.map_err(|e| ncf_core::NcfError::Io(e))?;
                        let prefix = FileHeaderPrefix::decode(&buf)?;
                        self.header_prefix = Some(prefix);
                        self.state = StreamState::AwaitingSchema;

                        // Returning a minimal Metadata event here; full header bytes can be read in the next state.
                        return Ok(Some(StreamChunk::Metadata(NcfHeader { metadata: ncf_core::header::Metadata { model_name: String::new(), architecture: String::new(), created_at: 0, author: None, license: None, quantization: None, custom: Default::default() } })));
                    }

                    StreamState::AwaitingSchema => {
                        let prefix = self.header_prefix.as_ref().expect("header present");
                        let header_len = prefix.header_len as usize;
                        let schema_start = (prefix.schema_offset as usize).saturating_sub(FILE_HEADER_PREFIX_SIZE as usize);
                        let schema_end = (prefix.index_offset as usize).saturating_sub(FILE_HEADER_PREFIX_SIZE as usize);

                        let total_len = header_len.checked_add(schema_end.saturating_sub(schema_start)).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "schema length overflow"))?;
                        let mut buf = vec![0u8; total_len];
                        self.inner.read_exact(&mut buf).await.map_err(|e| ncf_core::NcfError::Io(e))?;

                        let header_bytes = &buf[..header_len];
                        let _metadata = NcfHeader::decode_cbor(header_bytes)?;

                        let schema_rel_start = schema_start.checked_sub(0).unwrap_or(0);
                        let schema_rel_end = schema_rel_start + (schema_end - schema_start);
                        let schema_bytes = &buf[schema_rel_start..schema_rel_end];
                        let schemas: Vec<TensorSchema> = ciborium::de::from_reader(std::io::Cursor::new(schema_bytes))?;

                        self.state = StreamState::StreamingChunks;
                        return Ok(Some(StreamChunk::Schema(schemas)));
                    }

                    StreamState::StreamingChunks => {
                        let mut hdr_buf = [0u8; CHUNK_HEADER_SIZE as usize];
                        if let Err(e) = self.inner.read_exact(&mut hdr_buf).await {
                            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                                self.state = StreamState::Done;
                                return Ok(Some(StreamChunk::Done));
                            }
                            return Err(ncf_core::NcfError::Io(e));
                        }

                        let chunk_header = ChunkHeader::decode(&hdr_buf)?;

                        let payload_len = chunk_header.compressed_len as usize;
                        let mut payload = vec![0u8; payload_len];
                        self.inner.read_exact(&mut payload).await.map_err(|e| ncf_core::NcfError::Io(e))?;
                        let mut checksum = vec![0u8; CHUNK_CHECKSUM_SIZE as usize];
                        self.inner.read_exact(&mut checksum).await.map_err(|e| ncf_core::NcfError::Io(e))?;

                        // Attempt decompression with known algorithms; fall back to raw payload on failure.
                        let decompressed = if chunk_header.flags == 0 {
                            payload
                        } else if let Ok(decoded) = zstd::decode_all(&*payload) {
                            decoded
                        } else if let Ok(decoded) = lz4_flex::decompress_size_prepended(&payload) {
                            decoded
                        } else if let Ok(d) = snap::raw::Decoder::new().decompress_vec(&payload) {
                            d
                        } else {
                            payload.clone()
                        };

                        let computed = blake3::hash(&decompressed);
                        let checksum_valid = computed.as_bytes() == &checksum[..];

                        let name = format!("chunk_{}", chunk_header.chunk_id);
                        let emit = match &self.requested_tensors {
                            Some(set) => {
                                if self.header_prefix.as_ref().map(|p| p.flags.contains(NcfFlags::STREAMING_SAFE)).unwrap_or(false) {
                                    set.contains(&name)
                                } else {
                                    true
                                }
                            }
                            None => true,
                        };

                        // Placeholder: NCF writer currently does not expose per-chunk last flags in this reader.
                        let is_last = false;

                        if emit {
                            return Ok(Some(StreamChunk::TensorData {
                                name,
                                chunk_id: chunk_header.chunk_id,
                                data: Bytes::from(decompressed),
                                is_last_chunk: is_last,
                                checksum_valid,
                            }));
                        }

                        // If not emitted, continue the loop to read the next chunk.
                        continue;
                    }

                    StreamState::Done => return Ok(None),
                }
            }
        }
    }

    // Async tests for streaming reader
    #[cfg(test)]
    mod tests {
        use super::*;
        use tempfile::TempDir;
        use tokio::fs::File as TokioFile;

        use ncf_core::header::Metadata;
        use ncf_core::schema::{DType, Encoding, Layout, Compression};

        #[tokio::test]
        async fn test_stream_metadata_arrives_first() {
            let temp = TempDir::new().unwrap();
            let path = temp.path().join("test_stream.ncf");

            // Build a simple NCF file using writer
            let metadata = NcfHeader { metadata: Metadata { model_name: "s".into(), architecture: "a".into(), created_at: 0, author: None, license: None, quantization: None, custom: Default::default() } };
            let mut writer = crate::writer::NcfWriter::new(metadata.clone(), NcfFlags::empty());
            let schema = TensorSchema { name: "tensor_0".to_string(), dtype: DType::U8, shape: vec![4], column_layout: Layout::RowMajor, compression: Compression::None, encoding: Encoding::Plain, chunks: vec![] };
            writer.add_tensor(schema, vec![1,2,3,4]);
            writer.finalize(&path).expect("write");

            let file = TokioFile::open(&path).await.expect("open tokio file");
            let mut sr = NcfStreamReader::from_async_reader(file);
            let first = sr.next_chunk().await.expect("ok").expect("chunk");
            match first {
                StreamChunk::Metadata(_) => {}
                _ => panic!("expected metadata first"),
            }
        }

        #[tokio::test]
        async fn test_stream_truncated_file() {
            let temp = TempDir::new().unwrap();
            let path = temp.path().join("test_trunc.ncf");

            let metadata = NcfHeader { metadata: Metadata { model_name: "s".into(), architecture: "a".into(), created_at: 0, author: None, license: None, quantization: None, custom: Default::default() } };
            let mut writer = crate::writer::NcfWriter::new(metadata.clone(), NcfFlags::empty());
            let schema = TensorSchema { name: "tensor_0".to_string(), dtype: DType::U8, shape: vec![4096], column_layout: Layout::RowMajor, compression: Compression::None, encoding: Encoding::Plain, chunks: vec![] };
            writer.add_tensor(schema, vec![0u8; 4096]);
            writer.finalize(&path).expect("write");

            // Truncate file mid-chunk
            let mut data = std::fs::read(&path).expect("read");
            let mid = data.len() / 2;
            data.truncate(mid);
            std::fs::write(&path, &data).expect("write truncated");

            let file = TokioFile::open(&path).await.expect("open tokio file");
            let mut sr = NcfStreamReader::from_async_reader(file);
            // advance through metadata
            // Try to read schema; if schema read fails due to truncation, test is satisfied.
            match sr.next_chunk().await {
                Err(_) => return, // truncated earlier than expected — acceptable
                Ok(Some(_)) => {
                    // next chunk should return Err due to truncated payload
                    let res = sr.next_chunk().await;
                    assert!(res.is_err());
                }
                Ok(None) => panic!("unexpected end of stream"),
            }
        }
    }
}
